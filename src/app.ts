import cluster from 'cluster';
import os from 'os';
import express from 'express';
import multer from 'multer';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import fs from 'fs';
import path from 'path';
import logger from './logger';

const numCPUs = os.cpus().length;

if (cluster.isMaster) {
    logger.info(`Master ${process.pid} is running`);

    // Fork workers
    for (let i = 0; i < numCPUs; i++) {
        cluster.fork();
    }

    cluster.on('exit', (worker) => {
        logger.error(`Worker ${worker.process.pid} died`);
    });
} else {
    const app = express();
    const port = 3000;
    const upload = multer({ dest: 'uploads/' });

    let model: any;

    async function initialize() {
        model = await mobilenet.load();
        logger.info('Model loaded');
    }

    initialize();

    app.get('/', (req, res) => {
        res.send('Hello, World!');
    });

    app.post('/upload', upload.single('image'), async (req, res) => {
        if (!model) {
            return res.status(503).send('Server is still initializing. Please try again later.');
        }

        if (!req.file) {
            return res.status(400).send('No file uploaded');
        }

        const imgPath = path.join(__dirname, '..', req.file.path);
        const imageBuffer = fs.readFileSync(imgPath);
        let imageTensor = tf.node.decodeImage(imageBuffer, 3);

        imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
        console.log('Resized Shape:', imageTensor.shape);

        const predictions = await model.classify(imageTensor);

        imageTensor.dispose();
        fs.unlinkSync(imgPath);

        res.json({ predictions });
    });

    app.listen(port, () => {
        logger.info(`Worker ${process.pid} running on http://localhost:${port}`);
    });
}
