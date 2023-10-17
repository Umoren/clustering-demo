# Image Recognition API

## Overview

This project is an image recognition API built with Node.js and TensorFlow.js. It uses MobileNet for image classification and is designed to be scalable across multiple CPUs.

## Features

- **Scalable**: Utilizes all CPU cores for better performance.
- **Caching**: Implements LRU caching to store recent classifications.
- **Logging**: Detailed logging for debugging and performance metrics.
- **Error Handling**: Graceful error handling for better user experience.

## Requirements

- Node.js
- npm
- TensorFlow.js

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/image-recognition-api.git
cd image-recognition-api
```

Install the dependencies:

```bash
npm install
```

## Usage

To start the server, run:

```bash
npm start
```

The server will start on `http://localhost:3000`.

### API Endpoints

- **POST `/upload`**: Upload an image for classification.

## Performance

The API is designed to be efficient and has the following performance metrics:

- **Average Latency**: ~1.77 seconds
- **Max Latency**: ~3.7 seconds
- **Requests per Second**: ~5.54

## Future Improvements

- Implement load balancing for better scalability.
- Optimize the MobileNet model for faster classifications.
- Add asynchronous processing for heavy tasks.