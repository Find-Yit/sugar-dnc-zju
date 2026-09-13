import zipfile, sys, time, os
src = "/scratch/e1351071/zju_test/data/tandt_db.zip"
dst = "/scratch/e1351071/zju_test/data"
t0 = time.time()
with zipfile.ZipFile(src) as z:
    names = z.namelist()
    print(f"entries={len(names)}", flush=True)
    print("top-level:", sorted(set(n.split('/')[0] for n in names)), flush=True)
    z.extractall(dst)
print(f"done in {time.time()-t0:.1f}s", flush=True)
