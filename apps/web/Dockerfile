FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
COPY apps/web/package.json apps/web/package.json
COPY apps/api/package.json apps/api.package.json
COPY packages/shared/package.json packages/shared/package.json
# Copy repository contents so workspace file: dependencies resolve during install
COPY . .
RUN npm install
RUN npm run build -w packages/shared && npm run build -w apps/web
EXPOSE 3000
CMD ["npm", "run", "start", "-w", "apps/web"]
