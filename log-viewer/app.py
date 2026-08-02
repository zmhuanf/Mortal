import gzip
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOG_ROOT = r'D:/Workspace/Mortal/mortal/1v3/log'
PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def list_dirs():
    if not os.path.isdir(LOG_ROOT):
        return []
    return sorted((d for d in os.listdir(LOG_ROOT) if os.path.isdir(os.path.join(LOG_ROOT, d))), reverse=True)


def list_games(dir_name):
    dir_path = os.path.join(LOG_ROOT, dir_name)
    if not os.path.isdir(dir_path):
        return []
    return sorted(f for f in os.listdir(dir_path) if f.endswith('.json.gz'))


def load_actions(dir_name, game_name):
    with gzip.open(os.path.join(LOG_ROOT, dir_name, game_name), 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def game_display(name):
    parts = name[:-len('.json.gz')].split('_')
    return name if len(parts) != 3 else f'{parts[2]} · seed {parts[0]}'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        if urlparse(self.path).path in ('/', '/index.html'):
            self.serve_index()
        else:
            super().do_GET()

    def serve_index(self):
        params = parse_qs(urlparse(self.path).query)
        dirs = list_dirs()
        if not dirs:
            self.send_error(500, f'日志目录为空或不存在: {LOG_ROOT}')
            return
        cur_dir = params.get('dir', [dirs[0]])[0]
        if cur_dir not in dirs:
            cur_dir = dirs[0]
        games = list_games(cur_dir)
        cur_game = params.get('game', [games[0] if games else ''])[0]
        if not games:
            self.send_error(500, f'该目录下没有牌谱: {cur_dir}')
            return
        if cur_game not in games:
            cur_game = games[0]
        actions = load_actions(cur_dir, cur_game)

        with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f:
            html = f.read()
        dir_options = ''.join(
            f'<option value="{d}"{" selected" if d == cur_dir else ""}>{d}</option>' for d in dirs
        )
        game_options = ''.join(
            f'<option value="{g}"{" selected" if g == cur_game else ""}>{game_display(g)}</option>' for g in games
        )
        actions_json = json.dumps(actions, ensure_ascii=False).replace('</', '<\\/')
        html = (html.replace('__DIR_OPTIONS__', dir_options)
                    .replace('__GAME_OPTIONS__', game_options)
                    .replace('__ALL_ACTIONS__', actions_json))
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))


def main():
    print(f'1v3 牌谱浏览器: http://127.0.0.1:{PORT}')
    print(f'日志目录: {LOG_ROOT}')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
