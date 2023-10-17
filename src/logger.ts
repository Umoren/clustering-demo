import winston from 'winston';
import colors from 'colors';

const logger = winston.createLogger({
    format: winston.format.printf(({ level, message }) => {
        switch (level) {
            case 'info':
                return colors.green(`[INFO] ${message}`);
            case 'warn':
                return colors.yellow(`[WARN] ${message}`);
            case 'error':
                return colors.red(`[ERROR] ${message}`);
            default:
                return `[${level.toUpperCase()}] ${message}`;
        }
    }),
    transports: [
        new winston.transports.Console(),
    ],
});

export default logger;
