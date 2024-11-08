FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY ./package*.json ./
RUN npm install

# Copy project files
COPY . .

# Build the application
RUN npm run build

# Expose the Next.js port
EXPOSE 3000

CMD ["npm", "start"]