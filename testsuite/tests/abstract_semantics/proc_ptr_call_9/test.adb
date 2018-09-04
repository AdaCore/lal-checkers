procedure Test is
   P : access function return Integer;
   X : Integer := P.all;
begin
   null;
end Test;
