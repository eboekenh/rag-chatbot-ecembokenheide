# Ollama Guard — PreToolUse hook (Windows)
# Blocks shell commands involving Ollama when Ollama is not running on port 11434.

$raw = [Console]::In.ReadToEnd()
try {
    $data    = $raw | ConvertFrom-Json
    $command = $data.toolInput.command

    if ($command -match "ollama|qa_generator_ollama|test_ollama") {
        $client = New-Object System.Net.Sockets.TcpClient
        try {
            $client.Connect("localhost", 11434)
            $running = $true
        } catch {
            $running = $false
        } finally {
            $client.Close()
        }

        if (-not $running) {
            @{
                stopReason = "Ollama is not running on port 11434. Start it first: ollama serve"
            } | ConvertTo-Json
            exit 2
        }
    }
} catch {}
exit 0
