import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite 的配置文件:告诉它"这是一个 React 项目",这样它才知道怎么处理
// .jsx 文件(浏览器原生并不认识 JSX 语法,需要先被"翻译"成普通 JS)。
export default defineConfig({
  plugins: [react()],
});
