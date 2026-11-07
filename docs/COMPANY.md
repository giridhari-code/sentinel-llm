# Orbit — Company readiness

## Honest status

| Layer | Status |
|-------|--------|
| Architecture (Sentinel + Tesla) | Prototype complete |
| Train / eval / checkpoint | Works for small runs |
| Bhutu HTTP API | MVP (`sentinel serve`) |
| CI | GitHub Actions |
| Production GPU | Tesla-2 scaffold only |
| Auth / multi-tenant / billing | Not started |
| Safety filters | Not started |

**Company-ready** here means *early technical MVP foundation*, not full enterprise product.

## MVP checklist

- [x] Named product stack (Orbit / Sentinel / Tesla / DoBu / Bhutu)
- [x] Checkpoint save/load
- [x] Eval + perplexity
- [x] Generate with `--load`
- [x] Minimal HTTP API
- [x] CI (check + test + release build)
- [ ] Real trained weights in `models/`
- [ ] Auth + rate limits
- [ ] GPU production path
- [ ] Monitoring / metrics
- [ ] Legal data + model card

## Run the MVP

```bash
cargo run -p sentinel-infer -- train --steps 50 --save models/sentinel-small.ckpt
cargo run -p sentinel-infer -- generate --load models/sentinel-small.ckpt --prompt "Hello"
cargo run -p sentinel-infer -- serve --bind 127.0.0.1:8787 --load models/sentinel-small.ckpt
curl http://127.0.0.1:8787/health
curl -X POST http://127.0.0.1:8787/v1/generate -d "Hello Orbit"
```
