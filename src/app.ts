import cluster from 'cluster';
import os from 'os';
import express from 'express';
import multer from 'multer';
import * as tf from '@tensorflow/tfjs-node';
import * as mobilenet from '@tensorflow-models/mobilenet';
import fs from 'fs';
import path from 'path';
import logger from './logger';
import { LRUCache } from 'lru-cache'


const numCPUs = os.cpus().length;
const options = {
    max: 500, // The maximum number of items to store in the cache
    maxAge: 1000 * 60 * 30 // Items expire after an hour
};
const classificationCache = new LRUCache(options);


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
        const start = Date.now();
        if (!model) {
            return res.status(503).send('Server is still initializing. Please try again later.');
        }

        if (!req.file) {
            return res.status(400).send('No file uploaded');
        }

        const imgPath = path.join(__dirname, '..', req.file.path);
        const imageBuffer = await fs.promises.readFile(imgPath);

        const imageId = req.file.filename;

        const cachedResult = classificationCache.get(imageId);
        if (cachedResult) {
            return res.json({ predictions: cachedResult });
        }

        let imageTensor = tf.node.decodeImage(imageBuffer, 3);

        try {
            const tfStart = Date.now();
            imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
            const predictions = await model.classify(imageTensor);
            const tfEnd = Date.now();

            classificationCache.set(imageId, predictions);

            logger.warn(`Total time: ${Date.now() - start}ms, TensorFlow time: ${tfEnd - tfStart}ms`);
            res.json({ predictions });
        } catch (error) {
            logger.error(`Error during prediction: ${error}`);
            res.status(500).send('An error occurred during prediction.');
        } finally {
            imageTensor.dispose();
            await fs.promises.unlink(imgPath);
        }
    });



    app.listen(port, () => {
        logger.info(`Worker ${process.pid} running on http://localhost:${port}`);
    });
}
