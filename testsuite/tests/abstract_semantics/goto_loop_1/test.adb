procedure Test is
   X : Integer := 0;
begin
   <<loop1>>
   if X < 1000000 then
      X := X + 1;
      goto loop1;
   end if;
end Test;
