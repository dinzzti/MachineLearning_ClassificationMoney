"""
Jalankan script ini SEKALI sebelum menjalankan app.py:
    python patch_model.py

Script ini akan membuat file model_patched.keras yang bisa diload dengan benar.
"""
import zipfile, json, base64, marshal

src = "model_hybrid_terbaik.keras"
dst = "model_patched.keras"

# Bytecode baru: lambda x: tf.math.l2_normalize(x, axis=1)
# dengan 'import tensorflow as tf' di dalam fungsi (tidak butuh tf dari scope luar)
def _new_lambda(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis=1)

NEW_CODE_B64 = base64.b64encode(marshal.dumps(_new_lambda.__code__)).decode("utf-8")

print(f"Membaca {src}...")
with zipfile.ZipFile(src, 'r') as z:
    config_str = z.read('config.json').decode('utf-8')

config = json.loads(config_str)
layers = config['config']['layers']

patched = 0
for layer in layers:
    if layer.get('class_name') == 'Lambda':
        print(f"  Patching Lambda layer: {layer['config']['name']}")
        # Ganti bytecode dengan versi yang import tf secara internal
        layer['config']['function']['config']['code'] = NEW_CODE_B64
        # Tambahkan output_shape eksplisit agar Keras bisa infer shape
        layer['config']['output_shape'] = (2240,)
        patched += 1

if patched == 0:
    print("Tidak ada Lambda layer ditemukan.")
else:
    print(f"  {patched} Lambda layer berhasil dipatch.")

new_config = json.dumps(config).encode('utf-8')

print(f"Menyimpan ke {dst}...")
with zipfile.ZipFile(src, 'r') as zin:
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == 'config.json':
                zout.writestr(item, new_config)
            else:
                zout.writestr(item, zin.read(item.filename))

print(f"Selesai! File {dst} siap digunakan.")