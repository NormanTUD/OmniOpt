#!/usr/bin/perl

use strict;
use warnings;
use Term::ANSIColor;
use Data::Dumper;
use IO::Socket::INET;

my $intendation = 0;
my $debug = 0;
sub debug (@);

my $projectname = shift @ARGV;
my $name = shift @ARGV // 0;
my $slurmid = shift @ARGV // 0;

main($projectname, $name, $slurmid);

sub main {
    $intendation++;
    my $project = shift // "!!! No Project given !!!";
    my $start = shift;
    my $slurmid = shift;
    my $slurm_nodes = '127.0.0.1';
    if(exists $ENV{'SLURM_JOB_NODELIST'}) {
            $slurm_nodes = $ENV{'SLURM_JOB_NODELIST'};
    }

    debug "Slurm-Nodes: $slurm_nodes";

    print "GET_DMESG: main($project, $start, $slurmid)\n";
    my @server = get_servers_from_SLURM_NODES($slurm_nodes);

    foreach my $server (@server) {
        my $logpath = "/var/log/";
        my @logfiles = ("dmesg", "nhc.log");
        foreach my $this_logfile (@logfiles) {
            my $logfile = "$logpath$this_logfile";
            my $log_folder = "./projects/$project/logs/$slurmid/$this_logfile";
            system("mkdir -p $log_folder");
            my $log_file_name = "$server-$this_logfile.$name.log";
            my $log_file_path = "$log_folder/$log_file_name";

            my $command = "scp $server:/$logfile $log_file_path";
            print "$command\n";
            qx($command);
        }
    }
    $intendation--;
}

sub get_servers_from_SLURM_NODES {
    $intendation++;
    my $string = shift;
    debug "get_servers_from_SLURM_NODES($string)";
    my @server;
    while ($string =~ m#(.*?)\[(.*?)\](?:,|\R|$)#gi) {
            my $servercluster = $1;
            my $servernumbers = $2;
            foreach my $thisservernumber (split(/,/, $servernumbers)) {
                if($servernumbers !~ /-/) {
                    push @server, "$servercluster$thisservernumber";
                }
                if($servernumbers =~ m#(\d+)-(\d+)#) {
                    push @server, map { "$servercluster$_" } $1 .. $2;
                }
            }

            if($servernumbers =~ m#(\d+)-(\d+)#) {
                push @server, map { "$servercluster$_" } $1 .. $2;
            }
    }

    $intendation--;
    if(@server) {
        return @server;
    } else {
        return ("127.0.0.1");
    }
}

sub debug (@) {
        my @msg = @_;

        if($debug) {
            foreach (@msg) {
                warn(("\t" x $intendation).color("white")."$_".color("reset")."\n");
            }
        }
}
