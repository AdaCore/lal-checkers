procedure Test is
   x : Integer;
   y : Integer;
begin
   case x is
      when 2 .. 4 | 6 .. 8 =>
         y := 42;
      when others =>
         y := x;
   end case;
end Ex1;
