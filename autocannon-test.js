const autocannon = require('autocannon');
const fs = require('fs');

const boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW';
const filePath = '/Users/mac/Desktop/sammy23/image-recognition-api/image1.png';
const fileContent = fs.readFileSync(filePath);

const body = Buffer.concat([
    Buffer.from(
        `--${boundary}\r\n` +
        'Content-Disposition: form-data; name="image"; filename="image1.png"\r\n' +
        'Content-Type: image/png\r\n' +
        '\r\n'
    ),
    fileContent,
    Buffer.from(`\r\n--${boundary}--\r\n`)
]);

const instance = autocannon({
    url: 'http://localhost:3000/upload',
    method: 'POST',
    connections: 10,
    duration: 60,
    headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`
    },
    body: body
}, (err, result) => {
    if (err) {
        console.error('Error running autocannon:', err);
    } else {
        console.log('Test completed:', result);
    }
});

autocannon.track(instance, { renderProgressBar: true });
