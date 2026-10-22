# launchd agents (02 §6)
Install (the human decides whether to enable them; they use the whole CPU for hours):
    cp ops/launchd/com.vgcjax.t2.plist ops/launchd/com.vgcjax.t3.plist ~/Library/LaunchAgents/
    launchctl load ~/Library/LaunchAgents/com.vgcjax.t2.plist ~/Library/LaunchAgents/com.vgcjax.t3.plist
T2: nightly 02:00 (20k battles, mask checks on 10%). T3: Sunday 03:00 (100k battles + re-validation + bench + slow tests).
Both write reports/LATEST.md. Override size/stage with N=... STAGE=... in the plist's command.
