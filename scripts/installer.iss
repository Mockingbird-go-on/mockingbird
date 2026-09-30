; Inno Setup script for Mockingbird (Windows).
; Builds a single installer with CUDA support and automatic CPU fallback.
;
; Prerequisites: run scripts/build_windows.ps1 first (produces dist\mockingbird\).
; Then compile this script with ISCC.exe.
;
; User data (~/.mockingbird) is NEVER removed automatically; the uninstaller
; offers an optional cleanup task instead.

; All relative paths below are anchored to the PROJECT ROOT (a Windows-native
; project dir in the standard layout): {#SourcePath} is the scripts\ directory where
; this .iss lives.
; SourcePath = "<project>\scripts\" (WITH trailing backslash, verified
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
; Branding: left wizard banner (164x314, 24-bit uncompressed BMP — Inno
; does NOT scale wizard images, exact dimensions are mandatory).
WizardImageFile={#SourcePath}\wizard-side.bmp
; Small brand image in the top-right corner of every wizard page (55x58,
; 24-bit BMP; flattened onto the dark brand background — BMP has no alpha).
WizardSmallImageFile={#SourcePath}\wizard-small.bmp
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
; Two languages: Russian (primary, UI is Russian-first) and English. With two
; entries Inno's ShowLanguageDialog=auto shows the picker — useful for the
; growing EN audience. CustomMessages below provide the cyrillic data-dir
; prompt in Russian (English users never see it because data dir is in
; USERPROFILE, where paths are always ANSI/UTF-16-safe).
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Languages\English.isl"

[CustomMessages]
; Cyrrillic data-dir prompt + button labels. Inno substitutes the language
; prefix automatically (russian.* for the Russian installer, english.* for
; English). The English version is intentionally short — paths in the message
; are locale-neutral.
; CustomMessages keys must exist for EVERY [Languages] entry, otherwise Inno
; raises an unresolved-symbol error at compile time.
DeleteData=Mockingbird stores user data (resume, logs, knowledge base, profiles, history) in '%userprofile%\.mockingbird'. Also delete it?
DataKeep=Keep data
DataDelete=Delete everything
russian.DeleteData=Mockingbird хранит данные (резюме, логи, база знаний, профили, история) в '%userprofile%\.mockingbird'. Удалить их тоже?
russian.DataKeep=Сохранить данные
russian.DataDelete=Удалить всё
english.DeleteData=Mockingbird stores user data (resume, logs, knowledge base, profiles, history) in '%userprofile%\.mockingbird'. Also delete it?
english.DataKeep=Keep data
english.DataDelete=Delete everything

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

// -- Dark wizard theme (partial) -------------------------------------------
// Inno Setup has no native dark mode; we repaint what Pascal Script can
// reach: page backgrounds, header panel, label fonts and the hidden bevels.
// Buttons/checkboxes stay system-drawn (Win32 common controls) — they render
// as light islands, which is acceptable. All colors are BGR (Win32 COLORREF):
//   BrandDark  = #1A1D21 (app dark-theme surface)
//   BrandDark2 = #16181C (surface_alt)
//   TextMain   = #d1d9e1 (app text)
//   TextMuted  = #8a99a8 (app text_secondary)
const
  BrandDark = $211D1A;
  BrandDark2 = $1C1816;
  TextMain = $E1D9D1;
  TextMuted = $A8998A;

procedure _DarkenLabel(L: TNewStaticText);
begin
  if L = nil then exit;
  L.Color := BrandDark;
  L.Font.Color := TextMain;
end;

// Win32 imports for the progress-bar recolour (PBM_* work only after the
// comctl6 theme is stripped from the gauge — otherwise the theme paints
// the bar green and ignores the custom colours).
// NB: Inno Pascal has NO HWND/UINT/WPARAM/LPARAM/LRESULT types — Longint
// everywhere (same trap as the documented LPARAM issue below).
function SendMessage(hWnd: Longint; Msg: Longint; wParam: Longint; lParam: Longint): Longint;
  external 'SendMessageW@user32.dll stdcall';
procedure SetWindowTheme(hWnd: Longint; pszSubAppName: Longint; pszSubIdList: Longint);
  external 'SetWindowTheme@uxtheme.dll stdcall';

const
  PBM_SETBARCOLOR = $0409;
  PBM_SETBKCOLOR = $2001;
  BrandAccent = $1A2AFF; // BGR of #ff2a1a

procedure _DarkenTree(Root: TControl);
var
  I: Integer;
  Owner2: TWinControl;
begin
  // Recurse into every nested container (OuterNotebook/InnerNotebook hold
  // the per-page controls — a flat WizardForm.Controls walk never reaches
  // them, which left the page labels with black system text on dark bg).
  // NB: Inno's Pascal controls expose only a SUBSET of VCL properties —
  // TEdit/TMemo/TNewButton have no published Color here. We only touch what
  // the Support Classes Reference guarantees (TNewStaticText.Color, Font).
  if Root is TWinControl then
  begin
    Owner2 := TWinControl(Root);
    for I := 0 to Owner2.ControlCount - 1 do
      _DarkenTree(Owner2.Controls[I]);
  end;
  if Root is TNewStaticText then
    _DarkenLabel(TNewStaticText(Root));
end;

procedure InitializeWizard();
begin
  with WizardForm do
  begin
    Color := BrandDark;
    // Header strip (page name/description live here).
    MainPanel.Color := BrandDark;
    PageNameLabel.Color := BrandDark;
    PageNameLabel.Font.Color := TextMain;
    PageDescriptionLabel.Color := BrandDark;
    PageDescriptionLabel.Font.Color := TextMuted;
    // Welcome page big labels are reached through the recursive walk too,
    // but painted explicitly for clarity.
    WelcomeLabel1.Color := BrandDark;
    WelcomeLabel1.Font.Color := TextMain;
    WelcomeLabel2.Color := BrandDark;
    WelcomeLabel2.Font.Color := TextMuted;
    // Inner page surface.
    InnerPage.Color := BrandDark;
    // Tasks page («Дополнительные значки: Создать значок на Рабочем столе»):
    // the list itself is a TNewCheckListBox; both Color and Font are
    // published on it, so the row text goes light on dark.
    TasksList.Color := BrandDark2;
    TasksList.Font.Color := TextMain;
    // Finished page («Запустить Mockingbird»): the run checkbox sits on the
    // FinishedPage — repaint its surface dark and the caption light.
    FinishedPage.Color := BrandDark;
    RunList.Color := BrandDark;
    RunList.Font.Color := TextMain;
    // «Установка завершена» heading labels on the finished page.
    FinishedLabel.Font.Color := TextMain;
    FinishedHeadingLabel.Font.Color := TextMain;
    // Installing page («Распаковка файлов»): strip the comctl6 theme from
    // the gauge so PBM_SETBARCOLOR/SETBKCOLOR take effect, then paint the
    // bar brand-red on the dark track.
    SetWindowTheme(ProgressGauge.Handle, 0, 0);
    SendMessage(ProgressGauge.Handle, PBM_SETBARCOLOR, 0, BrandAccent);
    SendMessage(ProgressGauge.Handle, PBM_SETBKCOLOR, 0, BrandDark2);
    // Remove the light 3D separators — they glare against the dark bg.
    Bevel.Visible := False;
    Bevel1.Visible := False;
    // Walk the WHOLE control tree (nested notebooks included) and repaint
    // every TNewStaticText (page labels, dir-labels, disk-space captions…).
    _DarkenTree(WizardForm);
  end;
end;

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
      // CustomMessage references: Inno resolves {cm:DeleteData} / {cm:DataKeep}
      // / {cm:DataDelete} against the active language (russian.* or english.*)
      // automatically. Hard-coded Cyrillic in [Code] would show up in the
      // English installer too.
      RemoveData := MsgBox(
        ExpandConstant('{cm:DeleteData}') + #13#10 +
        DataDir + #13#10#13#10 +
        ExpandConstant('{cm:DataKeep}') + ' / ' + ExpandConstant('{cm:DataDelete}'),
        mbConfirmation, MB_YESNO) = IDYES;
      if RemoveData then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
