; Inno Setup script for Mockingbird (Windows).
; Builds a single installer with CUDA support and automatic CPU fallback.
;
; Prerequisites: run scripts/build_windows.ps1 first (produces dist\mockingbird\
; and dist\mockingbird-cli\). Then compile this script with ISCC.exe.
;
; User data (~/.mockingbird) is NEVER removed automatically; the uninstaller
; offers an optional cleanup task instead.

; All relative paths below are anchored to the PROJECT ROOT (E:\mockingbird
; in the standard layout): {#SourcePath} is the scripts\ directory where
; this .iss lives.
; SourcePath = "E:\mockingbird\scripts\" (WITH trailing backslash, verified
; against ISCC 6.7.3). Two RPos cuts drop the trailing delimiter and the
; "scripts" component, yielding the project root with a trailing backslash.
#define RootDir Copy(SourcePath, 1, RPos("\", Copy(SourcePath, 1, Len(SourcePath)-1)) - 1) + "\"
#define MyAppName "Mockingbird"
#define MyAppVersion GetVersionNumbersString(RootDir + "dist\mockingbird\mockingbird.exe")
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
OutputDir={#RootDir}installer
OutputBaseFilename=Mockingbird-Setup-{#MyAppVersion}
; {#SourcePath} = directory of this .iss file (works no matter what the
; current directory is when ISCC is invoked).
SetupIconFile={#SourcePath}\logo_mockingbird.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "cliicon"; Description: "Create Start Menu shortcut for mockingbird-cli (headless REPL)"; GroupDescription: "Additional icons:"

[Files]
Source: "{#RootDir}dist\mockingbird\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Per-user data dir is created at runtime, not by the installer.

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{group}\Mockingbird CLI"; Filename: "{app}\{#MyAppCliExeName}"; Tasks: cliicon; Parameters: "--cli"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Aggressive cleanup BEFORE file deletion:
;   (1) kill any running Mockingbird process (without this the uninstaller
;       cannot remove mockingbird.exe — "some elements could not be
;       removed");
;   (2) strip system/hidden/read-only attributes from {app} so Windows-
;       owned files like desktop.ini / folder.ico do not survive.
; The cmd ``2>nul & exit /b 0`` keeps uninstall going even if there is no
; running instance (taskkill returns 128 then).
Filename: "{cmd}"; Parameters: "/C taskkill /F /IM mockingbird.exe /T 2>nul & exit /b 0"; Flags: runhidden; RunOnceId: "killmockingbird"
Filename: "{cmd}"; Parameters: "/C attrib -S -H -R ""{app}\*.*"" /S /D 2>nul & exit /b 0"; Flags: runhidden; RunOnceId: "stripappattrs"

[UninstallDelete]
; Inno only removes what it installed; runtime artifacts (logs, __pycache__,
; crash dumps) created under {app} after install would otherwise leave an
; "could not remove some elements" notice. The app keeps ALL user data in
; ~/.mockingbird (handled by the dialog below), so wiping {app} on
; uninstall is safe.
Type: filesandordirs; Name: "{app}"

[Code]
const
  DataDirName = '.mockingbird';

function InitializeSetup(): Boolean;
begin
  Result := True;
end;

// Strip read-only / hidden / system attributes from every file under
// {app} BEFORE Inno runs its own file delete. Windows sometimes marks
// desktop.ini / folder.ico RO, and the install-time "ignoreversion"
// flag on those files does not help at uninstall time — Inno skips
// them silently and the user sees "some elements could not be removed".
// Done via [UninstallRun] (attrib -S -H -R /S /D) — Inno's Pascal
// Exec() signature is fragile across versions, and a dedicated
// UninstallRun entry is what every Inno docs example uses.
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
