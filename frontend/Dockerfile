# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code. `.dockerignore` keeps the host's node_modules and dist out:
# native bindings are platform-specific, so copying them over the alpine install
# breaks the build that follows.
COPY . .

# Vite inlines `import.meta.env.VITE_*` at BUILD time, so this has to arrive as
# a build argument. Supplying it as a runtime `environment:` entry on an nginx
# container serving a static bundle did nothing: whatever was in the committed
# `frontend/.env` got baked in instead, which meant every deployment pointed at
# http://localhost:8000.
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Build for production
RUN npm run build

# Stage 2: Serve with lightweight nginx
FROM nginx:alpine

# Copy built static files from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config (optional but recommended)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
