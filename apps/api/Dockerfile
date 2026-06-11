FROM node:22-alpine AS deps
WORKDIR /app
COPY package*.json ./
COPY apps/api/package.json apps/api/package.json
COPY apps/web/package.json apps/web/package.json
COPY packages/shared/package.json packages/shared/package.json
# Copy the repository contents so workspace file: dependencies resolve during install
COPY . .
RUN npm install
RUN npm run build -w packages/shared && npm run build -w apps/api
EXPOSE 4000
CMD ["npm", "run", "start", "-w", "apps/api"]
