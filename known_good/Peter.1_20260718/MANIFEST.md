# Known-good deployment snapshot: Peter.1

- Source: `C:\Users\admin\Desktop\Peter.1`
- Snapshot date: 2026-07-18
- Purpose: preserve the version explicitly accepted by the project owner before the current offline optimization work.
- Device state at snapshot export: the MaixCAM is currently powered off, so this snapshot is not being claimed as a fresh byte-for-byte read from the board.
- Model: `snail_eggs_yolo11n_320x224.mud`
- Tracking mode: the accepted stable deployment configuration, retained unchanged in this folder.
- Companion model binary: the desktop backup contained the MUD descriptor but not its referenced CVIMODEL. The matching generic CVIMODEL from the 2026-07-17 release package is included so this snapshot is deployable; its provenance and hash are recorded below.

## SHA256

```text
70536BC3DD57C0994031E50F86981B2F471131821A6983E6CFD5F216B692E196  maixcam/main.py
2B304A432BC96367B3AD956B010152A96C94841BB331B6DCE834D33CBCB16278  maixcam/web_control.py
92123D742911ACD14FF59D89D55FF6EC79ABEF1320C7AE64BC8039982508911F  README.md
E73B914445C6ACFC25F7EA77DCC09CAC5655CD98A7E0620B24744D37438610DE  root/models/snail_eggs_yolo11n_320x224.mud
E90D7F894E53BD48C0963C0D4AD527992D95B82DC56D0731EDD7895907A6D4E5  root/models/snail_eggs_yolo11n_320x224.cvimodel
9CC434BED7CAC1CE117BEBE111C7BAD81CF29E9DA864AA087494A287E1CAAF8E  scripts/maix_deploy_run.ps1
15F8757E41682D76CF5638157B9C13A2DFC3C7DF46A6779E8E9FB6C2FCBC4969  scripts/maix_remote.py
```

To restore this snapshot manually, copy its contents to the MaixCAM deployment locations described in the included README. Do not mix its model with the newer v26 candidate unless you intentionally want to test the candidate.
