# Define the sigma and theta values for WormCount
$sigma_theta_pairs = @{
    0.3 = @(0.031, 0.0315, 0.032, 0.0325)
    0.6 = @(0.038, 0.0385, 0.039, 0.0395)
    1.0 = @(0.0405, 0.041, 0.0415, 0.042)
}

# Define the maximum number of concurrent jobs, leaving 4 processors free
$totalProcessors = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
$maxJobs = [math]::Max(1, $totalProcessors - 4)  # Ensure at least 1 job can run

# Create all possible pairs of sigma and theta values
foreach ($sigma in $sigma_theta_pairs.Keys) {
    foreach ($theta in $sigma_theta_pairs[$sigma]) {
        # Wait until there is a free slot for another job
        while ((Get-Job | Where-Object { $_.State -eq 'Running' }).Count -ge $maxJobs) {
            Start-Sleep -Seconds 1
        }

        # Start a new job for each pair
        Start-Job -ScriptBlock {
            param($sigma, $theta, $scriptDir)

            # Run your program
            & "$scriptDir\sametheta_csd_2D.exe" $sigma $theta
        } -ArgumentList $sigma, $theta, $PSScriptRoot
    }
}

# Wait for all jobs to complete
Get-Job | Wait-Job

# Clean up the job list
Get-Job | Remove-Job