# Dev commands, run from the Mac. Everything heavy runs on the Lightning Studio.
STUDIO := godseye-studio
SSH := ssh -F infra/ssh_config
REMOTE := godsEye
PC := ~/opt/bin/process-compose --unix-socket ~/data/pc.sock 2>/dev/null

.PHONY: ssh sync bootstrap up down status logs tunnel publish-pass set-anthropic-key meva-spike meva-progress fake-live fake-live-stop fake-live-status meva-proxies meva-proxies-progress

# MEVA spike slice: school site, 16:15-16:45 (cameras start at 16:15 that hour).
MEVA_ARGS := --date 2018-03-11 --start 16:15 --end 16:45 --site school
FAKE_CAMS ?= G330 G339 G421 G639

ssh:
	$(SSH) $(STUDIO)

sync:
	rsync -az --delete --exclude-from=.rsyncignore -e "$(SSH)" \
	  --rsync-path="command -v rsync >/dev/null || sudo apt-get install -y -q rsync >/dev/null; rsync" \
	  ./ $(STUDIO):$(REMOTE)/

bootstrap: sync
	$(SSH) $(STUDIO) 'bash $(REMOTE)/infra/studio/bootstrap.sh && bash $(REMOTE)/infra/studio/install_on_start.sh'

up: sync
	$(SSH) $(STUDIO) 'bash $(REMOTE)/infra/studio/start.sh'

down:
	$(SSH) $(STUDIO) '$(PC) down'

status:
	$(SSH) $(STUDIO) '$(PC) process list'

logs:
	$(SSH) $(STUDIO) 'tail -n 50 ~/data/logs/process-compose.log'

# Browser -> API/HLS/WebRTC on localhost only; phone on the same Wi-Fi -> Mac :1935 -> Studio.
# Lightning's SSH gateway drops long sessions now and then; the loop reconnects (Ctrl-C stops it).
tunnel:
	@while true; do \
	  $(SSH) -N -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
	    -L 127.0.0.1:8000:localhost:8000 \
	    -L 127.0.0.1:8888:localhost:8888 \
	    -L 127.0.0.1:8889:localhost:8889 \
	    -L 127.0.0.1:8189:localhost:8189 \
	    -L 0.0.0.0:1935:localhost:1935 \
	    $(STUDIO); \
	  echo "$$(date +%H:%M:%S) tunnel closed; reconnecting in 3 s (Ctrl-C to stop)"; sleep 3; \
	done

# Copies the MediaMTX publish password (for Larix) to the clipboard without printing it.
publish-pass:
	@$(SSH) $(STUDIO) 'cat ~/data/secrets/mediamtx_publish_pass' | tr -d '\n' | pbcopy && echo "publish password copied to clipboard"

meva-spike: sync
	$(SSH) $(STUDIO) 'nohup python3 $(REMOTE)/scripts/meva/download.py $(MEVA_ARGS) > ~/data/logs/meva-download.log 2>&1 &' && echo "downloading; progress: make meva-progress"

meva-progress:
	@$(SSH) $(STUDIO) 'head -1 ~/data/logs/meva-download.log; tail -1 ~/data/logs/meva-download.log'

# Loops MEVA clips as live cameras: watch at http://localhost:8889/meva-<CAM>/ (WebRTC) or :8888 (HLS).
fake-live: sync
	$(SSH) $(STUDIO) 'bash $(REMOTE)/scripts/meva/fake_live.sh start $(FAKE_CAMS)'

fake-live-stop:
	$(SSH) $(STUDIO) 'bash $(REMOTE)/scripts/meva/fake_live.sh stop'

fake-live-status:
	@$(SSH) $(STUDIO) 'bash $(REMOTE)/scripts/meva/fake_live.sh status'

# 720p viewing copies (fake-live cameras first, then the rest of the slice), in the background.
meva-proxies: sync
	$(SSH) $(STUDIO) 'nohup sh -c "bash $(REMOTE)/scripts/meva/make_proxies.sh $(FAKE_CAMS) && bash $(REMOTE)/scripts/meva/make_proxies.sh" > ~/data/logs/meva-proxies.log 2>&1 &' && echo "converting; progress: make meva-proxies-progress"

meva-proxies-progress:
	@$(SSH) $(STUDIO) 'echo "$$(ls ~/data/meva/proxy/*.mp4 2>/dev/null | grep -vc tmp) of $$(ls ~/data/meva/raw/*.avi | wc -l) clips converted"; tail -1 ~/data/logs/meva-proxies.log'

# Stores the Anthropic key on the Studio (overrides Lightning's secret, no restart). Prompts here, hidden.
set-anthropic-key: sync
	@$(SSH) -t $(STUDIO) 'bash $(REMOTE)/infra/studio/set_anthropic_key.sh'
