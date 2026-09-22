; Inno Setup script for Mockingbird (Windows).
; Builds a single installer with CUDA support and automatic CPU fallback.
;
; Prerequisites: run scripts/build_windows.ps1 first (produces dist\mockingbird\).
; Then compile this script with ISCC.exe.
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

; Optional build variant (cuda|cpu), passed by build_windows.ps1 as
; /DBuildVariant=<variant>. It is baked into the output filename so the GPU
; and CPU installers coexist in installer\ instead of overwriting each other
; (both used to compile to Mockingbird-Setup-<ver>.exe). Invoking ISCC
; without the define (e.g. by hand) yields the generic name.
#ifdef BuildVariant
  #define VariantSuffix "-" + BuildVariant
#else
  #define VariantSuffix ""
#endif

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
OutputBaseFilename=Mockingbird-{#MyAppVersion}-windows-x64{#VariantSuffix}-setup
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
; Russian only: the app UI is Russian, so a language-selection wizard page is
; pure friction. With a single language entry Inno's ShowLanguageDialog=auto
; default skips the dialog entirely (it only appears for 2+ languages).
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#RootDir}dist\mockingbird\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Offline bundle: the zip ships a cache/ directory (HuggingFace snapshot of
; the default whisper model) NEXT TO setup.exe. Copy it straight into the
; user's model dir so the first launch does not download ~1.6 GB. This is an
; external file (read from {src} at install time, not compiled into setup),
; and skipifsourcedoesntexist makes a plain online installer (no cache/)
; silently ignore the entry instead of erroring. uninsneveruninstall keeps
; the model out of Inno's uninstall manifest: user data under ~/.mockingbird
; is only ever removed via the uninstall dialog (DelTree above).
Source: "{src}\cache\*"; DestDir: "{%USERPROFILE}\.mockingbird\models"; Flags: external ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist uninsneveruninstall

[Dirs]
; Per-user data dir is created at runtime, not by the installer.

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
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
; (3) Backup rmdir — Inno's RemoveDir({app}, True) only deletes files it
;     knows about (registered in unins000.dat). Anything that landed in
;     _internal\ after install (or that Inno's recursion missed) can
;     leave the {app} folder behind as a "dirty remainder". The
;     ping -n 5 gives Inno + the killed process ~4 s to fully release
;     every DLL mapping (cuDNN/cuBLAS/nvrtc especially hang on for
;     seconds after TerminateProcess); rmdir /S /Q then sweeps whatever
;     RemoveDir missed. Run as a separate cmd invocation so Inno itself
;     is fully gone by the time rmdir runs.
Filename: "{cmd}"; Parameters: "/C ping -n 5 127.0.0.1 >NUL & rmdir /S /Q ""{app}"" 2>nul & exit /b 0"; Flags: runhidden; RunOnceId: "sweepapp"

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

// The app stores its data in Path.home()/.mockingbird == %USERPROFILE%\.mockingbird
// (NOT %APPDATA%); the uninstaller offers to delete it, defaulting to keep.
//
// Before touching anything we best-effort stop a running Mockingbird so the
// SQLite database is not locked when DelTree runs. Inno's Pascal Script does
// NOT expose EnumWindows / LPARAM / Toolhelp32 (those types and functions do
// not exist — using them aborts the compiler with "Unknown type 'LPARAM'"),
// so a polite WM_CLOSE enumeration is impossible. The app autosaves its state
// on exit, and taskkill releases the file handles. The same kill is repeated
// by the [UninstallRun] "killmockingbird" entry as a fallback.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
  RemoveData: Boolean;
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    // 1) Make sure the app is closed BEFORE Inno touches {app} or the user
    //    data. Without this, locked files (mockingbird.exe,
    //    _internal\ctranslate2\*.dll) survive RemoveDir and leave
    //    "C:\Program Files\Mockingbird" with a partial set of files.
    //    `2>nul & exit /b 0` keeps uninstall going if it is not running.
    Exec(ExpandConstant('{cmd}'),
         '/C taskkill /F /T /IM mockingbird.exe 2>nul & exit /b 0',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Sleep(500);

    // 2) Ask about the real data directory.
    DataDir := ExpandConstant('{%USERPROFILE}') + '\' + DataDirName;
    if DirExists(DataDir) then
    begin
      RemoveData := MsgBox(
        'Удалить пользовательские данные (база знаний, профили, настройки, история)?' + #13#10 +
        DataDir + #13#10#13#10 +
        'Нажмите «Нет», чтобы сохранить данные для будущей переустановки.',
        mbConfirmation, MB_YESNO) = IDYES;
      if RemoveData then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
