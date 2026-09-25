#!/usr/bin/env node
// LoveFlow CLI - 全局命令入口
const { spawnSync } = require("child_process")
const path = require("path")
const fs = require("fs")

function findNativeBinary() {
  const home = process.env.HOME || process.env.USERPROFILE || ""
  const candidates = [
    path.join(home, ".loveflow", "bin", "loveflow"),
    path.join(home, ".loveflow", "bin", "mimo"),
    path.join(home, ".mimocode", "bin", "mimo"),
    process.env.MIMOCODE_BIN_PATH,
  ].filter(Boolean)
  for (const p of candidates) {
    if (fs.existsSync(p)) return p
    if (fs.existsSync(p + ".exe")) return p + ".exe"
  }
  return null
}

// 检查预编译二进制是否存在
const native = findNativeBinary()
if (native) {
  const r = spawnSync(native, process.argv.slice(2), { stdio: "inherit", shell: true })
  process.exit(r.status ?? 0)
}

// 尝试从项目源码用 bun 运行
const rootDirs = [
  path.join(__dirname, ".."),                                  // bin/..
  path.join(__dirname, "..", "..", ".."),                       // packages/opencode -> 项目根
  process.cwd(),
]
for (const dir of rootDirs) {
  const pkg = path.join(dir, "package.json")
  if (!fs.existsSync(pkg)) continue
  try {
    const { name } = JSON.parse(fs.readFileSync(pkg, "utf-8"))
    if (name === "@loveflow-ai/cli" || name === "loveflow-agent") {
      const r = spawnSync("bun", ["run", "--conditions=browser", "src/index.ts", ...process.argv.slice(2)], {
        cwd: dir,
        stdio: "inherit",
        shell: true,
        env: {
          ...process.env,
          PWD: process.env.PWD || process.cwd(),
          MIMOCODE_DISABLE_EXTERNAL_SKILLS: process.env.MIMOCODE_DISABLE_EXTERNAL_SKILLS || "1",
        },
      })
      if (!r.error) process.exit(r.status ?? 0)
    }
  } catch {}
}

// 检查是否需要先构建
const repoRoot = path.join(__dirname, "..", "..", "..")
const hasNodeModules = fs.existsSync(path.join(repoRoot, "node_modules"))
const hasBunBinary = fs.existsSync(path.join(repoRoot, "packages", "opencode", "bin", "mimo.cjs"))

console.error("")
console.error("╔══════════════════════════════════════════════════════════╗")
console.error("║           LoveFlow Code 需要使用说明                      ║")
console.error("╚══════════════════════════════════════════════════════════╝")
console.error("")
console.error("未检测到预编译 CLI 二进制文件。请选择以下方式之一启动：")
console.error("")
if (hasBunBinary && hasNodeModules) {
  console.error("方式一：手动使用 bun 开发模式（Windows 可能不稳定）")
  console.error("   bun run --cwd packages/opencode dev")
  console.error("")
}
console.error("方式一：运行 install 脚本安装预编译 CLI")
console.error("   curl -fsSL https://loveflow.dev/install | bash")
console.error("")
console.error("方式二：使用 npm 全局链接")
console.error("   cd packages/opencode && npm link")
console.error("   loveflow --help")
console.error("")
console.error("方式三：运行 bun run dev 启动开发模式")
console.error("   cd packages/opencode && bun run dev")
console.error("")
process.exit(1)
