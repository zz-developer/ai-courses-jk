import { readdirSync, existsSync, rmdirSync, mkdirSync, renameSync, writeFileSync } from "fs";
import { resolve } from "path";

const exceptFolders = ["shared", ".DS_Store"];

const dirs = readdirSync(resolve("courses")).filter((dir) => {
  return !exceptFolders.includes(dir);
});

let index = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Course List</title>
</head>

<body>
    <h1>Course List</h1>
    <ul>
`

dirs.forEach((dir) => {
  // Move `dist` in current folder to root `dist` folder with its name
  const distDir = resolve("dist");
  if (existsSync(distDir)) {
    rmdirSync(distDir, { recursive: true });
  }
  mkdirSync(distDir);
  const src = resolve("courses", dir, "dist");
  const dest = resolve("dist", dir);
  if (existsSync(src)) {
    mkdirSync(dest);
    const files = readdirSync(src);
    files.forEach((file) => {
      renameSync(resolve(src, file), resolve(dest, file));
    });
  }
  index += `        <li><a href="/${dir}">${dir}</a></li>\n`
});

index += `    </ul>

</body>
</html>
`

writeFileSync(resolve("dist", "index.html"), index);