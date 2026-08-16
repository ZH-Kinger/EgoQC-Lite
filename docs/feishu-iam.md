# 飞书 IAM 与人工复检任务分配

EgoQC 使用飞书 OAuth 识别复检人员，使用 PostgreSQL 保存内部角色、会话和视频任务归属。阿里云 IAM 不参与应用登录。

## 身份与权限

- 飞书 `open_id` 映射为稳定内部身份 `feishu:<open_id>`。姓名只用于显示，不用作唯一键。
- 首个登录用户自动成为 `admin`，后续用户默认为 `reviewer`。管理员可在 PostgreSQL 的 `review_users.role` 中调整角色。
- `reviewer` 只能看到并操作分配给自己的视频；`admin` 可看全局、执行自动分配。
- 同一视频的所有异常事件始终分配给同一人，避免上下文被拆散。
- 自动分配使用“异常片段时长 + 每事件 2 秒固定复检成本”估算负载，并按 `capacity_weight` 进行加权贪心均衡。

## 飞书应用配置

1. 在飞书开放平台创建企业自建应用，启用网页应用登录。
2. 配置重定向 URL：正式环境建议 `https://<review-domain>/auth/callback`。本地 SSH 隧道测试可使用 `http://127.0.0.1:8767/auth/callback`。
3. 配置应用可用范围，只包含数据复检人员。
4. 将 `deploy/feishu.env.example` 复制到部署机 `/srv/egoqc/secrets/feishu.env`，填写 App ID、App Secret 和重定向 URL，权限设为 `600`。
5. 重启 `egoqc-review` 服务。环境变量完整时自动启用飞书登录；三项都未配置时保持手填审核员的本地开发模式。

## 登录流程与安全

1. `/auth/login` 创建一次性 `state` 和 PKCE verifier，数据库仅保存 `state` 的 SHA-256。
2. 飞书回调 `/auth/callback` 后，使用授权码 + PKCE 交换 `user_access_token`，再读取用户信息。
3. 应用仅在登录时使用 access token，不将它持久化。本地 session token 仅以 SHA-256 形式入库，Cookie 使用 `HttpOnly; SameSite=Lax`，HTTPS 下自动加 `Secure`。
4. 可选 `FEISHU_ALLOWED_TENANT_KEYS` 会拒绝非指定飞书租户。

## 日常使用

1. 所有复检人先各自登录一次，建立用户记录。
2. 管理员点击“自动分配”，仅为尚未分配的视频建立归属，不改动已分配任务。
3. 复检人在“我的任务”中领取并提交。已处理卡片会从工作队列消失，但仍可在“已完成”中查询。
4. 新增视频进库后，管理员再执行一次“自动分配”即可滚动追加。
