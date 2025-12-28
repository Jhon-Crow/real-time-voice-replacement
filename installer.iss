; Inno Setup Script for Voice Replacer
; This script creates a Windows installer with proper Start Menu shortcuts,
; uninstaller, and optional desktop shortcut.

#define MyAppName "Voice Replacer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Voice Replacer Contributors"
#define MyAppURL "https://github.com/Jhon-Crow/real-time-voice-replacement"
#define MyAppExeName "VoiceReplacer.exe"

[Setup]
; Application information
AppId={{A7E8B9C2-D3F4-5E6A-7B8C-9D0E1F2A3B4C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; Output settings
OutputDir=dist
OutputBaseFilename=VoiceReplacer-{#MyAppVersion}-setup
; Compression
Compression=lzma2
SolidCompression=yes
; Installer appearance
WizardStyle=modern
; Windows version requirements
MinVersion=10.0
; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Privileges
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Uninstaller
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupicon"; Description: "Start {#MyAppName} with Windows"; GroupDescription: "Other:"

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startupicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Check if VB-Audio Virtual Cable is installed and show a message if not
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Show a message about VB-Audio Virtual Cable requirement
    if MsgBox('Voice Replacer requires VB-Audio Virtual Cable to work properly.' + #13#10 + #13#10 +
              'Would you like to open the VB-Audio website to download it?',
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      // Open VB-Audio website
      ShellExec('open', 'https://vb-audio.com/Cable/', '', '', SW_SHOW, ewNoWait, ResultCode);
    end;
  end;
end;

var
  ResultCode: Integer;
