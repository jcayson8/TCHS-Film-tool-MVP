FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY backend/package.json ./backend/
WORKDIR /app/backend
RUN npm install

# Copy the app source
WORKDIR /app
COPY backend ./backend

# Expose port and start
EXPOSE 8080
CMD ["node","backend/server.js"]
