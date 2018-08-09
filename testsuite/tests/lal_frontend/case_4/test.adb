procedure Test is
   x : Integer;
   y : Integer;
begin
   case x is
      when Natural =>
         y := 43;
      when others =>
         y := x;
   end case;
end Ex1;
