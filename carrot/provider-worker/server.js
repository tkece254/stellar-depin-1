const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' }));

const jobsDir = path.join(__dirname, 'jobs');
const resultsDir = path.join(__dirname, 'results');
if (!fs.existsSync(jobsDir)) fs.mkdirSync(jobsDir);
if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir);

const PINATA_JWT = process.env.PINATA_JWT;
const PINATA_API_URL = 'https://api.pinata.cloud/pinning/pinFileToIPFS';
const PINATA_JSON_URL = 'https://api.pinata.cloud/pinning/pinJSONToIPFS';

async function uploadImageToIPFS(filePath, filename) {
  if (!PINATA_JWT) { console.error('[Carrot] no pinata jwt'); return null; }

  try {
    const FormData = require('form-data');
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    formData.append('pinataMetadata', JSON.stringify({ name: filename }));

    const response = await axios.post(PINATA_API_URL, formData, {
      headers: { 'Authorization': `Bearer ${PINATA_JWT}`, ...formData.getHeaders() },
      maxBodyLength: Infinity,
    });

    console.log(`[Carrot] image pinned: ${response.data.IpfsHash}`);
    return response.data.IpfsHash;
  } catch (e) {
    console.error('[Carrot] ipfs upload failed:', e.message);
    return null;
  }
}

async function uploadMetadataToIPFS(metadata, filename) {
  if (!PINATA_JWT) return null;

  try {
    const response = await axios.post(PINATA_JSON_URL, {
      pinataContent: metadata,
      pinataMetadata: { name: filename },
    }, {
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${PINATA_JWT}` },
    });

    console.log(`[Carrot] metadata pinned: ${response.data.IpfsHash}`);
    return response.data.IpfsHash;
  } catch (e) {
    console.error('[Carrot] metadata upload failed:', e.message);
    return null;
  }
}

// the main event - where the magic happens
app.post('/process-job', async (req, res) => {
  const { jobId, jobType, jobData } = req.body;

  console.log(`\n${'='.repeat(50)}`);
  console.log(`[Carrot] Job #${jobId} incoming - lets go`);
  console.log(`[Carrot] Type: ${jobType}`);
  console.log(`${'='.repeat(50)}\n`);

  try {
    let result;

    if (jobType === 'python-script') {
      result = await runPythonScript(jobId, jobData);
    } else if (jobType === 'docker-image') {
      result = await runDockerImage(jobId, jobData);
    } else {
      throw new Error(`unknown job type: ${jobType}`);
    }

    console.log(`\n[Carrot] Job #${jobId} done - ez\n`);

    res.json({
      success: true,
      jobId,
      result: result.output,
      resultHash: result.hash,
      logs: result.logs
    });

  } catch (e) {
    console.error(`\n[Carrot] Job #${jobId} failed: ${e.message}\n`);
    res.status(500).json({ success: false, error: e.message, logs: e.logs || '' });
  }
});

// spin up python in a container
async function runPythonScript(jobId, scriptData) {
  return new Promise(async (resolve, reject) => {
    const scriptPath = path.join(jobsDir, `job_${jobId}.py`);
    fs.writeFileSync(scriptPath, scriptData.code);

    console.log(`[Carrot] saved script, running in docker...`);

    const isMac = process.platform === 'darwin';
    const gpuFlag = isMac ? '' : '--gpus all';

    const dockerCmd = `docker run --rm ${gpuFlag} \
      -v ${scriptPath}:/job.py:ro \
      -v ${resultsDir}:/results \
      --memory="2g" \
      --cpus="2" \
      carrot-gpu-worker \
      python3 /job.py`.replace(/\s+/g, ' ');

    if (isMac) console.log(`[Carrot] macos detected - cpu mode`);

    exec(dockerCmd, { maxBuffer: 10 * 1024 * 1024 }, async (error, stdout, stderr) => {
      fs.unlinkSync(scriptPath);

      if (error) {
        reject({ message: 'script go boom', logs: stderr || stdout });
        return;
      }

      const resultFile = path.join(resultsDir, `job_${jobId}_output.txt`);
      let fileOutput = '';
      if (fs.existsSync(resultFile)) {
        fileOutput = fs.readFileSync(resultFile, 'utf8');
      }

      const output = fileOutput || stdout.trim();

      // upload result to ipfs
      const metadata = {
        jobId,
        result: output,
        executionLogs: stdout,
        timestamp: new Date().toISOString(),
        platform: 'Carrot GPU DePIN'
      };

      const ipfsHash = await uploadMetadataToIPFS(metadata, `job-${jobId}-result.json`);
      const hash = ipfsHash ? `ipfs://${ipfsHash}` : `0x${Buffer.from(output).toString('hex').slice(0, 16)}`;

      resolve({ output, hash, logs: stdout });
    });
  });
}

// pull and run consumer's docker image
async function runDockerImage(jobId, imageData) {
  return new Promise((resolve, reject) => {
    let imageName = imageData.image;
    const imageArgs = imageData.args || '';

    // people always paste the url smh
    if (imageName.includes('hub.docker.com')) {
      const match = imageName.match(/hub\.docker\.com\/r\/([^\/]+\/[^\/\?]+)/);
      if (match) {
        imageName = match[1];
        console.log(`[Carrot] extracted image: ${imageName}`);
      }
    }

    console.log(`[Carrot] pulling: ${imageName}`);

    exec(`docker pull ${imageName}`, (pullError, pullStdout, pullStderr) => {
      if (pullError) {
        reject({ message: 'cant pull image', logs: pullStderr });
        return;
      }

      console.log(`[Carrot] pull done, running...`);

      const isMac = process.platform === 'darwin';
      const gpuFlag = isMac ? '' : '--gpus all';

      const jobOutputDir = path.join(resultsDir, `job_${jobId}_output`);
      if (!fs.existsSync(jobOutputDir)) fs.mkdirSync(jobOutputDir, { recursive: true });

      const dockerCmd = `docker run --rm ${gpuFlag} \
        -v ${jobOutputDir}:/output \
        --memory="4g" \
        --cpus="4" \
        ${imageName} ${imageArgs}`.replace(/\s+/g, ' ');

      if (isMac) console.log(`[Carrot] macos - cpu mode`);

      exec(dockerCmd, { maxBuffer: 10 * 1024 * 1024 }, async (error, stdout, stderr) => {
        if (error) {
          reject({ message: `container died: ${error.message}`, logs: `STDOUT:\n${stdout}\n\nSTDERR:\n${stderr}` });
          return;
        }

        const output = stdout.trim();
        let hash = `0x${Buffer.from(output).toString('hex').slice(0, 16)}`;

        const outputFiles = [];
        let imageIpfsHash = null;

        if (fs.existsSync(jobOutputDir)) {
          const files = fs.readdirSync(jobOutputDir);
          for (const file of files) {
            const filePath = path.join(jobOutputDir, file);
            const stats = fs.statSync(filePath);

            if (file.match(/\.(png|jpg|jpeg|gif|webp)$/i) && stats.size < 5 * 1024 * 1024) {
              console.log(`[Carrot] found output: ${file}`);

              imageIpfsHash = await uploadImageToIPFS(filePath, `job-${jobId}-${file}`);

              if (imageIpfsHash) {
                const metadata = {
                  jobId,
                  image: `ipfs://${imageIpfsHash}`,
                  imageFilename: file,
                  imageSize: stats.size,
                  executionLogs: stdout,
                  result: output,
                  timestamp: new Date().toISOString(),
                  platform: 'Carrot GPU DePIN'
                };

                await uploadMetadataToIPFS(metadata, `job-${jobId}-metadata.json`);
                hash = `ipfs://${imageIpfsHash}`;
              }

              outputFiles.push({
                filename: file,
                data: fs.readFileSync(filePath).toString('base64'),
                type: file.split('.').pop().toLowerCase(),
                size: stats.size,
                ipfsHash: imageIpfsHash
              });
            }
          }
        }

        resolve({
          output,
          hash,
          logs: stdout,
          files: outputFiles,
          ipfsImage: imageIpfsHash ? `ipfs://${imageIpfsHash}` : null
        });
      });
    });
  });
}

app.get('/health', (req, res) => {
  res.json({ status: 'running', gpuAvailable: true, jobsProcessed: 0 });
});

app.get('/check-docker', (req, res) => {
  exec('docker --version', (error, stdout) => {
    res.json(error ? { dockerInstalled: false } : { dockerInstalled: true, version: stdout.trim() });
  });
});

app.get('/check-gpu', (req, res) => {
  exec('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', (error, stdout) => {
    if (error) {
      res.json({ gpuAvailable: false });
    } else {
      const [name, memory] = stdout.trim().split(',');
      res.json({ gpuAvailable: true, name: name.trim(), memory: memory.trim() });
    }
  });
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log('\n  ___ __ _ _ __ _ __ ___ | |_');
  console.log(' / __/ _` | \'__| \'__/ _ \\| __|');
  console.log('| (_| (_| | |  | | | (_) | |_');
  console.log(' \\___\\__,_|_|  |_|  \\___/ \\__|');
  console.log('\n GPU Worker - Stellar Edition');
  console.log(`\n Server: http://localhost:${PORT}`);
  console.log(' Isolation: Docker containers');
  console.log(' Currency: XLM (lumens)');
  console.log('\n Waiting for jobs...\n');
});
