# Deployment Guide - PatrimonioSmart

## Quick Start with Railway

### 1. Prerequisites
- GitHub account
- Railway account (https://railway.app)
- Domain purchased (e.g., patrimoniosmart.com)

### 2. Push to GitHub

```bash
# Initialize git if needed
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for production deployment"

# Create repo on GitHub and push
git remote add origin https://github.com/YOUR_USERNAME/patrimoniosmart.git
git push -u origin main
```

### 3. Deploy to Railway

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway will auto-detect the Dockerfile

### 4. Add PostgreSQL

1. In your Railway project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will auto-set `DATABASE_URL`

### 5. Configure Environment Variables

In Railway dashboard → Variables, add:

```
DASHBOARD_AUTH_ENABLED=true
DASHBOARD_PASSWORD=your_secure_password
ANTHROPIC_API_KEY=your_key_here
SCHEDULER_ENABLED=true
LOG_LEVEL=INFO
```

### 6. Migrate Data (Optional)

If you have existing data in SQLite:

```bash
# Get PostgreSQL URL from Railway dashboard
python scripts/migrate_to_postgres.py \
    --source sqlite:///data/financial_data.db \
    --target "postgresql://user:pass@host:5432/railway"
```

### 7. Connect Custom Domain

1. In Railway → Settings → Domains
2. Click "Add Custom Domain"
3. Enter: `patrimoniosmart.com`
4. Add DNS records at your registrar:
   - Type: CNAME
   - Name: @ or www
   - Value: (Railway provides this)

### 8. SSL Certificate

Railway automatically provisions SSL certificates via Let's Encrypt.

---

## Project Structure for Production

```
patrimoniosmart/
├── Dockerfile           # Container configuration
├── railway.toml         # Railway settings
├── requirements.txt     # Python dependencies
├── .env.production.example
├── src/                 # Core application
├── web/
│   ├── app.py          # Streamlit dashboard
│   └── static/         # Images, CSS
└── scripts/
    └── migrate_to_postgres.py
```

---

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `DASHBOARD_AUTH_ENABLED` | Yes | Enable login (true/false) |
| `DASHBOARD_PASSWORD` | Yes | Login password |
| `ANTHROPIC_API_KEY` | No | For AI assistant |
| `SCHEDULER_ENABLED` | No | Auto-download data |
| `LOG_LEVEL` | No | INFO, DEBUG, WARNING |

---

## Monitoring

### Health Check
```
https://patrimoniosmart.com/_stcore/health
```

### Logs
```bash
railway logs
```

---

## Scaling

Railway auto-scales based on traffic. To manually adjust:

1. Railway Dashboard → Settings
2. Adjust "Compute" resources
3. Recommended: 1GB RAM minimum for this app

---

## Troubleshooting

### Database Connection Error
- Verify `DATABASE_URL` is set correctly
- Check Railway PostgreSQL is running

### Slow Loading
- Railway may sleep inactive apps (free tier)
- First request after sleep takes ~30s

### Auth Not Working
- Ensure `DASHBOARD_AUTH_ENABLED=true`
- Ensure `DASHBOARD_PASSWORD` is set

---

## Cost Estimate (Railway)

| Resource | Free Tier | Pro Tier |
|----------|-----------|----------|
| Compute | $5/month credit | Pay as you go |
| PostgreSQL | 500MB | Pay per GB |
| Bandwidth | 100GB | Pay per GB |

Estimated monthly cost: **$5-15/month**
