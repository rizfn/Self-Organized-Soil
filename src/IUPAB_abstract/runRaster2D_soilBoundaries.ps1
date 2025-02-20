# Define the sigma and theta values for WormCount
$sigma_values = @(0.1, 0.5, 1.0)
$theta_values = 0..39 | ForEach-Object { 0.08 + $_ * (0.16 - 0.08) / 39 }

# Define the new theta values by adding points in the middle of the current points
$theta_values_new = @()
for ($i = 0; $i -lt $theta_values.Count - 1; $i++) {
    $mid_point = ($theta_values[$i] + $theta_values[$i + 1]) / 2
    $theta_values_new += $mid_point
}


# Define the maximum number of concurrent jobs, leaving 4 processors free
$totalProcessors = (Get-WmiObject -Class Win32_ComputerSystem).NumberOfLogicalProcessors
$maxJobs = [math]::Max(1, $totalProcessors - 4)  # Ensure at least 1 job can run

# Create all possible pairs of sigma and theta values
foreach ($sigma in $sigma_values) {
    foreach ($theta in $theta_values_new) {
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
            & "$scriptDir\soil_boundaries_2D.exe" $sigma $theta
        } -ArgumentList $sigma, $theta, $PSScriptRoot
    }
}

# Wait for all jobs to complete
Get-Job | Wait-Job

# Clean up the job list
Get-Job | Remove-Job