# Skin Check API

1. `git init && git add . && git commit -m "init" && git branch -M main`
2. Create GitHub repo, `git remote add origin <url>`, `git push -u origin main`
3. Render: New > Web Service > connect repo (render.yaml is auto-detected)
4. Set `ALLOWED_ORIGINS` to your site URLs (comma-separated)
5. Test: `https://<your-app>.onrender.com/health`

Use from a site:
```js
const f = new FormData(); f.append("file", fileInput.files[0]);
const r = await fetch("https://<your-app>.onrender.com/api/check-skin", {method:"POST", body:f});
```
Verify that the model's sigmoid = p_non_skin against your training labels.
