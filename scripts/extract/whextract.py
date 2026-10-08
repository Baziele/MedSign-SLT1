"""Extract archive.zip with progress. Run detached on Windows box."""
import zipfile

Z = r"C:\Users\Baziele\Documents\Code\medsign-multiview\archive.zip"
D = r"C:\Users\Baziele\Documents\Code\medsign-multiview\data"

z = zipfile.ZipFile(Z)
names = z.namelist()
print(f"extracting {len(names)} entries...", flush=True)
for i, n in enumerate(names):
    z.extract(n, D)
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(names)}", flush=True)
print("EXTRACTION DONE", flush=True)
