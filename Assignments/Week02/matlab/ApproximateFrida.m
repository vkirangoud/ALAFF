IMG = imread( 'Frida.jpg' );    
A = double( IMG( :,:,1 ) );     
imshow( uint8( A ) )          
size( A )

[ U, Sigma, V ] = svd( A );

k = 1
B = uint8( U( :, 1:k ) * Sigma( 1:k,1:k ) * V( :, 1:k )' );   
imshow( B );
pause();

% Repeat this with increasing k.
%r = min( size( A ) );
%for k=1:r
%	imshow( uint8( U(:, 1:k ) * Sigma( 1:k, 1:k ) * V( :, 1:k )' ) );
%        input( strcat( num2str( k ), " press return" ) );
%end

%pause();

% To determine a reasonable value for k it helps to graph the singular values:
figure
r = min( size( A ) );
plot( [1:r ], diag( Sigma ), 'x' );
figure
loglog( [1:r ], diag( Sigma ), 'x' );

pause();
