# AI Project GitHub report
*Grup 3 (Nizam, Pathya, Fauzan)*

AI yang dapat mendeteksi sampah di kelas dengan kamera(WebCam/CCTV)

------------------------------Detail-------------------------------

**1. EMPATHIZE (Memahami Masalah & User)**
- *User*
  Semua pihak yang ada di sekolah
- *Yang dialami oleh user*
  sampah masih bergeletakan diatas maupun dibawah meja.
  kelas jadi berantakan dan kotor.
  Siswa lupa untuk membuang sampah pada tempatnya.
- *Kebutuhan user*
  Sistem otomatis yang bisa mendeteksi sampah yang ada di kelas.
  Pengingat otomatis kalau ada sampah yang belum dibuang.

**2. DEFINE (Merumuskan Masalah Utama)**
“Siswa terkadang lupa ataupun sengaja tidak membuang sampah pada tempatnya, alhasil staf harus lebih ekstra dalam beres beres, oleh karena itu Dibutuhkan solusi otomatis yang dapat mengidentifikasi, mencatat, dan mengingatkan peminjaman secara real-time.”

**3. IDEATE (Menciptakan Ide Solusi)**
  Potensi Solusi:
1. AI Computer Vision (CV)
   - Kamera di area kelas.
   - AI mendeteksi jenis sampah (botol, saus, tutup botol, dll ).
   - Sistem mengenali bentuk sampah.
2. AI Reminder System
   - Jika ada sampah yang belum dibuang maka kelas akan mendapatkan notifikasi.
3. Dashboard kelas
   - Menampilkan:
     item apa yang belum dibuang
4. Gamification for Students
   - Siswa dapat poin jika membuang sampah.
   - Ranking kebiasaan baik tingkat kelas.

**4. PROTOTYPE (Merancang Purwarupa)**
Tampilan Sistem (High-Level):
1. Area kelas dengan Kamera AI
   - Kamera Dinyalakan
   - AI mendeteksi "Siswa X membuang sampah +1 botol"
   - Terekam otomatis ke database
3. Data transfer untuk ke notifikasi
   - Data: nama sampah, kelas, jumlah sampah.
   - Setelah data terkumpul akan dijadikan sebuah teks pesan
5. Data menjadi notifikasi ke Gmail
   - Contoh pesan:
     Data Kelas Hari ini (T/B/T)
     Nama Sampah = [Sampah_saus, Sampah_kresek,...]
     Jumlah = (...)

**5. TEST (Pengujian & Feedback)**
Langkah uji coba:
- Pilih 10 siswa sebagai tester.
- Tes menaruh Di area Webcam → AI deteksi → catat otomatis → Pengiriman Data.
- Tes kasus khusus:
  bawa dua item sekaligus
  item tertutup tangan
  ruangan gelap

