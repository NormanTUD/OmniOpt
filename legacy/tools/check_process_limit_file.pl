#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

my $percent_limit = 0.9;

sub get_errors_from_file {
    my $filename = shift;
    my $hostname = shift;
    open my $fh, '<', $filename or die "$!";
    my @errors = ();
    my $i = 0;
    while (my $line = <$fh>) {
        if($i != 0) {
            my @splitted = map { chomp $_; $_ } split(",", $line);
            my $used_processes = $splitted[2];
            my $max_processes = $splitted[3];
            if($used_processes >= int($percent_limit * $max_processes)) {
                push @errors, "Almost reached limit of $max_processes ($used_processes used processes) on $hostname at time $splitted[0]. This may explain failing forks.\n";
            }
        }
        $i++;
    }
    close $fh;
    return \@errors;
}

my $logdir = shift @ARGV;

die "No logdir specified" unless $logdir;
die "Directory $logdir does not exist" unless -d $logdir;

my %check_files = ();

while (my $dir = <$logdir/*>) {
    if(-d $dir) {
        if($dir =~ m#process-check-(.*)#) {
            my $hostname = $1;
            my $file = "$dir/processcheck.csv";
            if(-e $file) {
                $check_files{$hostname} = {
                    file => $file,
                    errors => get_errors_from_file($file, $hostname)
                }
            }
        }
    }
}

my $num_errors = 0;
foreach my $host (keys %check_files) {
    my @errors = @{$check_files{$host}{errors}};
    if(@errors) {
        foreach my $error (@errors) {
            print "$error\n";
            $num_errors++;
        }
    }
}

exit $num_errors;
