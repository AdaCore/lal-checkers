procedure Ex1 is
   x : Integer;
   y : Integer;
begin
   case x is
      when 1 =>
         y := 12;
      when 2 | 4 =>
         y := 42;
      when 3 =>
         y := 3141592;
      when others =>
         y := x;
   end case;
end Ex1;
