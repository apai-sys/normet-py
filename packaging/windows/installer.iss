; Inno Setup script for the Windows normet installer.
; Compiled in CI by: ISCC.exe packaging\windows\installer.iss
; Expects PyInstaller output at dist\normet\ (onedir, built from
; packaging\normet_gui.spec).

; Version is injected by CI from pyproject.toml via ISCC /DAppVersion=…
; The fallback here only applies when compiling the script by hand.
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppName=Normet
AppVersion={#AppVersion}
AppPublisher=apai-sys
DefaultDirName={autopf}\Normet
DefaultGroupName=Normet
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=normet-setup-{#AppVersion}
SetupIconFile=..\assets\normet.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\..\dist\Normet\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Normet"; Filename: "{app}\Normet.exe"
Name: "{commondesktop}\Normet"; Filename: "{app}\Normet.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Normet.exe"; Description: "Launch Normet"; Flags: nowait postinstall skipifsilent
