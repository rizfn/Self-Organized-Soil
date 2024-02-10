# Define the number of elements
$elements = 20

# Define the sigma and theta values
$sigma_values = 0..($elements-1) | ForEach-Object { $_ / ($elements-1) }
$theta_values = 0..($elements-1) | ForEach-Object { $_ * 0.3 / ($elements-1) }

# Define the maximum number of concurrent jobs
$maxJobs = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors

# Create all possible pairs of sigma and theta values
foreach ($sigma in $sigma_values) {
    foreach ($theta in $theta_values) {
        # Wait until there is a free slot for another job
        while ((Get-Job | Where-Object { $_.State -eq 'Running' }).Count -ge $maxJobs) {
            Start-Sleep -Seconds 1
        }

        # Start a new job for each pair
        Start-Job -ScriptBlock {
            param($sigma, $theta, $scriptDir)

            # Run your program
            & "$scriptDir\nutrientTimeseries3D.exe" $sigma $theta
        } -ArgumentList $sigma, $theta, $PSScriptRoot
    }
}

# Wait for all jobs to complete
Get-Job | Wait-Job

# Clean up the job list
Get-Job | Remove-Job