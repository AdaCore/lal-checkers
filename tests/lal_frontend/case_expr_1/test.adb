procedure Ex1 is
   x : Integer;
   y : Integer;
begin
   y := case x is
        when 2 .. 4 | 6 .. 8 => 42,
        when 314 => 1592,
        when others => x;
end Ex1;
