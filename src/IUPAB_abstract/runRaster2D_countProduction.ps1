# Define the sigma and theta values for WormCount
$sigma_values = @(0.1, 0.5, 1.0)
$theta_values = 0..19 | ForEach-Object { 0.08 + $_ * (0.16 - 0.08) / 19 }

# Define the maximum number of concurrent jobs, leaving 4 processors free
$totalProcessors = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
$maxJobs = [math]::Max(1, $totalProcessors)  # Ensure at least 1 job can run

# Create all possible pairs of sigma and theta values
foreach ($sigma in $sigma_values) {
    foreach ($theta in $theta_values) {
        # Ignore theta = 0
        if ($theta -eq 0) {
            continue
        }

        # Wait until there is a free slot for another job
        while ((Get-Job | Where-Object { $_.State -eq 'Running' }).Count -ge $maxJobs) {
            Start-Sleep -Seconds 1
        }

        # Start a new job for each pair
        Start-Job -ScriptBlock {
            param($sigma, $theta, $scriptDir)

            # Run your program
            & "$scriptDir\nutrientTimeseries2D_countProduction.exe" $sigma $theta
        } -ArgumentList $sigma, $theta, $PSScriptRoot
    }
}

# Wait for all jobs to complete
Get-Job | Wait-Job

# Clean up the job list
Get-Job | Remove-Job