@echo off
cd /d C:\Users\usuario\financial-data-project
py -c "import sys; sys.path.insert(0, '.'); from src.yahoo_downloader import YahooDownloader; downloader = YahooDownloader(); downloader.download_all()"
