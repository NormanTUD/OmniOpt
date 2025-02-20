#!/usr/bin/perl

use strict;
use warnings;
use autodie;
use File::Basename;
use IO::Socket::INET;
use Carp;

#### This script allows to write an IP for the mongo-db-server
#### to the project's folder

my %options = (
        project => undef,
        ip => get_local_ip_address(),
        slurmid => undef
);

foreach (@ARGV) {
        if(/--project=(.*)/) {
                $options{project} = $1;
        } elsif (/--ip=(.*)/) {
                $options{ip} = $1;
        } elsif (/--slurmid=(.*)/) {
                $options{slurmid} = $1;
        } else {
                croak("Unknown parameter $_");
        }
}

croak "No project!" unless $options{project};
croak "No slurm-id!" unless $options{slurmid};

my $dirname = dirname(__FILE__).'/projects/'.$options{project};
croak "`$dirname` could not be found!" unless -d $dirname;

$dirname = "$dirname/ipfiles/";
mkdir $dirname unless -d $dirname;

my $file = "$dirname/mongodbserverip-$options{slurmid}";
print "Writing IP to $file\n";

if (-e $file) {
    carp "$file already exists!";
    #unlink $file or croak $!;
}

if(is_ipv4($options{ip})) {
    open my $fh, '>', $file or croak $!;
#    $options{ip} = "127.0.0.1";
    print $fh $options{ip};
    close $fh;
} else {
    croak("ERROR: $options{ip} is not a valid IPv4-address!");
}

# https://stackoverflow.com/questions/330458/how-can-i-determine-the-local-machines-ip-addresses-from-perl
sub get_local_ip_address {
    my $socket = IO::Socket::INET->new(
        Proto       => 'udp',
        PeerAddr    => '198.41.0.4', # a.root-servers.net
        PeerPort    => '53', # DNS
    );

    # A side-effect of making a socket connection is that our IP address
    # is available from the 'sockhost' method
    my $local_ip_address = $socket->sockhost;

    return $local_ip_address;
}

sub is_ipv4 {
    my $ip = shift;
    if ($ip =~ /^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/) {
        return 1;
    } else {
        return 0;
    }
}
