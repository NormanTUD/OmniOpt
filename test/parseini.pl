#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;

my $inifile = shift;

die Dumper read_ini_file($inifile);

sub read_ini_file { 
        my $file = shift;
        my $conf;
        open (my $INI, $file) || die "Can't open ini-file `$file`: $!\n";
        my $section = '';
        while (<$INI>) {
                chomp;
                if (/^\s*\[\s*(.+?)\s*\]\s*$/) {
                        $section = $1;
                }

                if ( /^\s*([^=]+?)\s*=\s*(.*?)\s*$/ ) {
                        $conf->{$section}->{$1} = $2;         
                }
        }
        close ($INI);
        return $conf;
}
