# CS2 Demo to 2D Replay (AWPy)

Bu proje, CS2 `.dem` dosyasindan AWPy ile detayli veri cikarip `replay_tool.html` icinde 2D replay olarak gosterir.

## Neler Cekiliyor?

- Oyuncu: isim, steamid, takim, pozisyon `(X, Y, Z)`, can, armor, silah, para (`cash` varsa)
- Olaylar: kill, damage, shot, smoke, inferno, bomb eventleri
- Bomba:
- Event bazli bomb kayitlari (`plant`, `defuse`, `explode`, vb.)
- Bomba tasiyici rotasi (`bomb_carrier_path`) eger `has_bomb` verisi parse edilirse
- Utility:
- Grenade trajeleri (`grenades`)
- Grenade hedef/inis noktasi (`grenade_landings`)

## Kurulum ve Veri Çıkarma

### Seçenek 1: Notebook (Önerilen - Kolay)

1. `demo_to_data.ipynb` dosyasını VS Code'da aç
2. Üst kısımda kernel seç (Python 3.x)
3. Hücre 1 → Hücre 2 → ... → sırayla çalıştır ("Run Cell" butonu)
4. Hücre 9'da `demo_data.json` otomatik oluşturulur

### Seçenek 2: Command Line (CLI)

```bash
py -m pip install awpy polars
py extract_demo_data.py vita-auro.dem -o demo_data.json --tick-sample 4 --grenade-sample 2
```

Parametreler:

- `--tick-sample`: her N tickten 1 kayit alir (dosya boyutunu dusurur)
- `--grenade-sample`: grenade trajelerini seyrekleştirir
- `--verbose`: AWPy parse loglarini acar

> Not: `python` yerine `py` kullan (Windows Python launcher)

## Replay Tool Kullanimi

1. `replay_tool.html` dosyasini tarayicida ac.
2. `demo_data.json` dosyasini yukle.
3. Round sec, tick slider ile gez, play tusu ile replay baslat.

## Yeni Eklenen Gorseller

- Bomba tasiyici rotasi: sari kesikli cizgi
- Grenade hedef noktasi: utility tipine gore renkli marker
- Event log tekrari engelleme: ayni tickte repaint oldugunda log spam yapmaz
- Round timeline paneli: round icindeki kill/bomb olaylarini listeler
- Oyuncu trail gorunumu: son birkac saniyelik hareket izi
- Opsiyonel map katmani: JSON icindeki `meta.map_image` varsa arkaplanda cizer

## Map Gorseli Kullanimi (Opsiyonel)

- Notebook export hucresinde `meta.map_image` alanina bir yol veya URL verebilirsin.
- Ornek: `maps/de_dust2.png`
- Replay icinde `Map` toggle ile acip kapatabilirsin.

## Notlar ve Sorun Giderme

- `cash` veya `has_bomb` gibi ekstra alanlar demo/awpy sürümüne göre olmayabilir.
- Script, ekstra alanlarla parse başarısız olursa temel player property listesi ile tekrar dener.

### "Python bulunamadı" hatası

- **Çözüm 1:** Notebook ortamı kullan (Seçenek 1 / kolay)
- **Çözüm 2:** `py` command'ı kullan (Windows Python launcher)
  ```bash
  py --version  # Test et
  py -m pip install awpy polars  # Paketi kur
  py extract_demo_data.py ...  # Scripti çalıştır
  ```
- **Çözüm 3:** Anaconda/Python düzgün kurulu değilse Anaconda Navigator açıp bir environment oluştur
