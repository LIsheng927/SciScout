import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

// 这几行是几乎每个 React 项目都一样的"启动代码":
// 找到 index.html 里那个 <div id="root">,把 <App /> 这个组件"挂"上去。
// 以后所有的界面逻辑,都写在 App.jsx 里,这个文件基本不用再改。
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
