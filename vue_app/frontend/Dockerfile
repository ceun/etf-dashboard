FROM node:22-alpine AS build

WORKDIR /app
COPY vue_app/frontend/package*.json ./
RUN npm install
COPY vue_app/frontend ./
ARG VITE_API_BASE_URL=/api
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY vue_app/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
