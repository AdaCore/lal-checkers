procedure Test is
   x : Integer;
begin
   goto Test;
   x := 2;
<<Test>>
   x := 1;
end Ex1;
