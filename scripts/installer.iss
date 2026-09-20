; Inno Setup script for Mockingbird (Windows).
; Builds a single installer with CUDA support and automatic CPU fallback.
;
; Prerequisites: run scripts/build_windows.ps1 first (produces dist\mockingbird\
; and dist\mockingbird-cli\). Then compile this script with ISCC.exe.
;
; User data (~/.mockingbird) is NEVER removed automatically; the uninstaller
; offers an optional cleanup task instead.

#define MyAppName "Mockingbird"
#define MyAppVersion GetVersionNumbersString("dist\mockingbird\mockingbird.exe")
#define MyAppPublisher "Mockingbird"
#define MyAppExeName "mockingbird.exe"
#define MyAppCliExeName "mockingbird-cli.exe"

[Setup]
AppId={{7C1B6D9E-2A55-4B7E-9A1F-0C3D5E8B9A42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=Mockingbird-Setup-{#MyAppVersion}
SetupIconFile=scripts\logo_mockingbird.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Large CUDA-torch bundle: raise the size checks so installs on full drives
; fail with a clear message instead of a mid-copy error.
ExtraDiskSpaceMB=5120

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "cliicon"; Description: "Create Start Menu shortcut for mockingbird-cli (headless REPL)"; GroupDescription: "Additional icons:"

[Files]
Source: "dist\mockingbird\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Per-user data dir is created at runtime, not by the installer.

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{group}\Mockingbird CLI"; Filename: "{app}\{#MyAppCliExeName}"; Tasks: cliicon; Parameters: "--cli"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Only remove build artifacts; user knowledge bases under {userdocs} stay.

[UninstallRun]
; Optional data cleanup is handled via the custom uninstall page below.

[Code]
const
  DataDirName = '.mockingbird';

function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
  RemoveData: Boolean;
begin
  if CurUninstallStep = usUninstall then
  begin
    // The app stores data in Path.home()/.mockingbird == %USERPROFILE%\.mockingbird
    // (NOT %APPDATA%). Ask about the real directory.
    DataDir := ExpandConstant('{%USERPROFILE}') + '\' + DataDirName;
    if DirExists(DataDir) then
    begin
      RemoveData := MsgBox(
        'Delete user data (knowledge bases, profiles, settings, SQLite history)?' + #13#10 +
        DataDir + #13#10#13#10 +
        'Choose No to keep your data for a future reinstall.',
        mbConfirmation, MB_YESNO) = IDYES;
      if RemoveData then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
