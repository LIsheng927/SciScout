# 构建阶段:装依赖、跑 vite build,产出静态文件
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

# 运行阶段:只把构建产物丢进一个轻量的 nginx 镜像,不带 node_modules 这些构建工具
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
