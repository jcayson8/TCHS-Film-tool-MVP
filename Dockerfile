FROM node:20-alpine

RUN apk add --no-cache ffmpeg

WORKDIR /app

# Install Node dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy backend source files
COPY server.js ./
COPY analysis.js ./
COPY teamIdentity.js ./

# Create the public frontend folder expected by server.js
RUN mkdir -p /app/public

# Copy frontend and logo
COPY index.html ./public/index.html
COPY Logo.PNG ./public/Logo.PNG

EXPOSE 8080

CMD ["node", "server.js"]
